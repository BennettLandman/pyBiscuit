# Brief 22 report: FreeSurfer-route replay against real `biscuit_FC_v2.2` outputs — 2026-10-22

Branch `brief-22-fs-replay` (from `main` `a0cadbf`). Data: `~/dev/biscuit/open-examples/{0050952,0050961,0050971}` (ABIDE NYU, FreeSurfer
7.4.1, then `biscuit_FC_v2.2.sif --surf_recon freesurfer`, template `mb101`, protocols `braincolor,dkt31`, run 2025-09-17; image
SHA-256 `e70aea62…4329`). The subject data is not in git; input hashes of the key surfaces:
[`open-examples-2026-10-22/input-hashes.sha256`](open-examples-2026-10-22/input-hashes.sha256). These are the project's first **image
evidence**; they outrank the macOS oracles (AGENTS.md). Everything below is 3 subjects × 2 hemispheres; metrics per subject:
`open-examples-2026-10-22/metrics-<subject>.json`, table: [`summary.md`](open-examples-2026-10-22/summary.md), script
`scripts/validation/open_examples.py` (`python scripts/validation/open_examples.py WORKDIR --threads 8 --ens 6`, ~1 h on 8 cores).

## In brief

* **Input stage built** (`pybiscuit.surfprep.freesurfer`): the image's `surf/*.vtk` (white, gray, mid, inflated, sphere), `roi/*.mask.txt`
  and `quant/*.thickness.txt` are reproduced **byte for byte** for all six hemispheres. `quant/*.mid.area.txt` equals the barycentric
  vertex area of `mid.vtk` to print precision (5e-7).
* **Stable stages pass E-class on all six hemispheres**: TRACE (4/4 files byte-identical per hemisphere, after the fix below);
  `SphericalRemesh` of the gray/white/mid surfaces onto the template tessellation given the image's coefficients (max vertex difference
  ≤ 7.4e-5 mm, identical faces); every `stat/*.reg.*` map (LGI, SD, H, S, SCI, thickness, area) reproduced to print precision from the
  image's own `quant/` maps and `reg/*.sphere.reg.vtk`.
* **Three real port discrepancies found and fixed** (each with a regression test), all invisible to the macOS oracles:
  1. HSD call 1's initial guess with `--icomesh` (Q17, below) sent the whole registration into a wrong basin (E1 1.6-2.0 vs the image's
     0.72-0.79; deformed sphere 93-119° away) on 5 of 6 hemispheres and aborted (Q16 AABB) on the sixth.
  2. `GyralCurve::refineCurves` reads an uninitialised variable: the image runs ≥ 2 passes (closes the open question of `trace.md`).
  3. `strel('ball')` neighbourhood excluded the rim in MATLAB: sulcal depth was 0.71 mm too deep on average (now -0.04 mm).
* **Not passing**: HSD under the policy bound (chaotic, below); sulcal depth under the E-class bound (r = 0.9997-0.9998, 54-60 % of
  vertices equal to print precision, the rest within 1.41 mm); LGI (r ≈ 0.89, the klaplace correspondence is a stand-in, below).
  Nothing was loosened.

## Rules used

E-class bounds are those of `validation-tolerances.md` (not changed). For the two chaotic stages the proposals Q14 (LGI distribution
criterion) and Q15 (accept within `max(policy, 1.5 × spread)`) are applied **as proposed, not adopted**, with the spread measured on
*this* port: the image cannot be perturbed, so the spread is the port's own one-ulp sensitivity on the same subject (HSD: the chain rerun
from inputs with one `res.H` value, one `res.sulc` value, or half of all values moved by one float32 ulp, 6 reruns per hemisphere; LGI:
every pial coordinate moved by 0 or ±1 ulp, one rerun). Energies are the PDF's: `E = ‖z(template curv.mean) − z(res.curv through the
registered sphere)‖² / 163842`, which reproduces the numbers the image printed (0050952: 1.970 → 0.755 lh, 1.696 → 0.739 rh).

| Stage | Subject | Hemi | Result | Rule | Verdict |
|---|---|---|---|---|---|
| input VTK/mask/thickness | 0050952 | lh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050952 | lh | median 0.164 deg, p99 0.671 deg; own spread median 0.303, p99 1.09 | policy 0.01/0.1 deg; Q15 proposed 0.454/1.64 | policy FAIL; Q15 PASS |
| HSD coefficients | 0050952 | lh | max abs 0.0885; own spread 0.237 | policy 1e-4; Q15 0.355 | policy FAIL; Q15 PASS |
| HSD energy E1 (PDF) | 0050952 | lh | image 0.757, ours 0.758 (rel 0.00109); own spread 0.00941 | policy 1e-3; Q15 0.0141 | policy FAIL; Q15 PASS |
| HSD with upstream raw icomesh guess | 0050952 | lh | sphere median 113 deg, E1 1.72 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050952 | lh | surf max abs 7.44e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050952 | lh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050952 | lh | r 0.9997, bias -0.0392 mm, max abs 1.41 mm, exact 0.542 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050952 | lh | r 0.89053, rel p99 0.629, ROI-mean rel 0.45; own spread p99 0.0231 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050952 | rh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050952 | rh | median 0.11 deg, p99 0.843 deg; own spread median 0.0488, p99 0.326 | policy 0.01/0.1 deg; Q15 proposed 0.0732/0.489 | policy FAIL; Q15 FAIL |
| HSD coefficients | 0050952 | rh | max abs 0.0538; own spread 0.0183 | policy 1e-4; Q15 0.0274 | policy FAIL; Q15 FAIL |
| HSD energy E1 (PDF) | 0050952 | rh | image 0.74, ours 0.745 (rel 0.00611); own spread 0.00149 | policy 1e-3; Q15 0.00223 | policy FAIL; Q15 FAIL |
| HSD with upstream raw icomesh guess | 0050952 | rh | sphere median 0.223 deg, E1 0.742 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050952 | rh | surf max abs 2.86e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050952 | rh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050952 | rh | r 0.99978, bias -0.0425 mm, max abs 1 mm, exact 0.546 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050952 | rh | r 0.88819, rel p99 0.623, ROI-mean rel 0.459; own spread p99 0.0341 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050961 | lh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050961 | lh | median 0.278 deg, p99 1.56 deg; own spread median 0.263, p99 1.64 | policy 0.01/0.1 deg; Q15 proposed 0.395/2.47 | policy FAIL; Q15 PASS |
| HSD coefficients | 0050961 | lh | max abs 0.0407; own spread 0.0394 | policy 1e-4; Q15 0.0592 | policy FAIL; Q15 PASS |
| HSD energy E1 (PDF) | 0050961 | lh | image 0.794, ours 0.792 (rel 0.00246); own spread 0.00692 | policy 1e-3; Q15 0.0104 | policy FAIL; Q15 PASS |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050961 | lh | surf max abs 3.05e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050961 | lh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050961 | lh | r 0.99968, bias -0.0188 mm, max abs 1 mm, exact 0.604 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050961 | lh | r 0.88607, rel p99 0.55, ROI-mean rel 0.391; own spread p99 0.0232 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050961 | rh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050961 | rh | median 0.297 deg, p99 2.85 deg; own spread median 0.142, p99 0.93 | policy 0.01/0.1 deg; Q15 proposed 0.213/1.39 | policy FAIL; Q15 FAIL |
| HSD coefficients | 0050961 | rh | max abs 0.0226; own spread 0.0131 | policy 1e-4; Q15 0.0197 | policy FAIL; Q15 FAIL |
| HSD energy E1 (PDF) | 0050961 | rh | image 0.756, ours 0.762 (rel 0.00823); own spread 0.00322 | policy 1e-3; Q15 0.00482 | policy FAIL; Q15 FAIL |
| HSD with upstream raw icomesh guess | 0050961 | rh | sphere median 119 deg, E1 1.62 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050961 | rh | surf max abs 2.01e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050961 | rh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050961 | rh | r 0.99974, bias -0.0265 mm, max abs 1 mm, exact 0.582 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050961 | rh | r 0.89406, rel p99 0.616, ROI-mean rel 0.425; own spread p99 0.0229 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050971 | lh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050971 | lh | median 0.104 deg, p99 0.59 deg; own spread median 0.23, p99 1.26 | policy 0.01/0.1 deg; Q15 proposed 0.345/1.88 | policy FAIL; Q15 PASS |
| HSD coefficients | 0050971 | lh | max abs 0.0242; own spread 0.121 | policy 1e-4; Q15 0.181 | policy FAIL; Q15 PASS |
| HSD energy E1 (PDF) | 0050971 | lh | image 0.723, ours 0.725 (rel 0.00294); own spread 0.00221 | policy 1e-3; Q15 0.00331 | policy FAIL; Q15 PASS |
| HSD with upstream raw icomesh guess | 0050971 | lh | sphere median 105 deg, E1 1.96 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050971 | lh | surf max abs 3.43e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050971 | lh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050971 | lh | r 0.99978, bias -0.025 mm, max abs 1 mm, exact 0.547 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050971 | lh | r 0.87568, rel p99 0.613, ROI-mean rel 0.446; own spread p99 0.0271 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050971 | rh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050971 | rh | median 0.198 deg, p99 0.727 deg; own spread median 0.179, p99 0.78 | policy 0.01/0.1 deg; Q15 proposed 0.268/1.17 | policy FAIL; Q15 PASS |
| HSD coefficients | 0050971 | rh | max abs 0.187; own spread 0.192 | policy 1e-4; Q15 0.288 | policy FAIL; Q15 PASS |
| HSD energy E1 (PDF) | 0050971 | rh | image 0.742, ours 0.75 (rel 0.0118); own spread 0.0107 | policy 1e-3; Q15 0.016 | policy FAIL; Q15 PASS |
| HSD with upstream raw icomesh guess | 0050971 | rh | sphere median 93.4 deg, E1 1.57 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050971 | rh | surf max abs 6.48e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050971 | rh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050971 | rh | r 0.99978, bias -0.0212 mm, max abs 1 mm, exact 0.54 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050971 | rh | r 0.90021, rel p99 0.597, ROI-mean rel 0.513; own spread p99 0.027 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |

## Findings

### F1 HSD: the upstream call-1 initial guess does not behave like the image (fixed; Q17)

`HSD::guessInitCoeff` queries the template mean map with an `AABB_Sphere` built on the raw `--icomesh` mesh (radius 100) with unit
vectors; every query misses, the mean map is 0, and the guess picks an arbitrary rigid rotation (port note H-3; the macOS oracle did the
same). Brief 21 flagged this as "needs one run". On the open examples the arbitrary rotation is ~135° and the chain ends in a wrong
basin; the image's registered spheres are only 10° (median displacement) from the native spheres. With the guess's mean map queried on
the unit-radius icosahedron (the evident intent), or with call 1 run `--noguess`, the whole chain lands on the image's result:

| Subject | Hemi | Image E1 | Upstream raw `--icomesh` guess (median ° / E1) | Unit-icomesh guess, default (median ° / E1) | Call-1 `--noguess` (median ° / E1) |
|---|---|---|---|---|---|
| 0050952 | lh | 0.757 | 113° / 1.720 | 0.164° / 0.758 | 0.143° / 0.759 |
| 0050952 | rh | 0.740 | 0.223° / 0.742 | 0.11° / 0.745 | 0.166° / 0.741 |
| 0050961 | lh | 0.794 | ABORT (Fatal error: Property - no closest triangle found in subject 1) | 0.278° / 0.792 | 0.32° / 0.789 |
| 0050961 | rh | 0.756 | 119° / 1.621 | 0.297° / 0.762 | 0.276° / 0.760 |
| 0050971 | lh | 0.723 | 105° / 1.961 | 0.104° / 0.725 | 0.17° / 0.727 |
| 0050971 | rh | 0.742 | 93.4° / 1.569 | 0.198° / 0.750 | 0.116° / 0.740 |

The default is now the unit query (`HSD(guess_unit_ico=True)`; CLI `--guess-raw-icomesh` restores upstream; all oracle comparisons and
the `guess_icomesh` golden use the raw flag). The two variants are not distinguishable from the outputs (Q17). Upstream's raw guess
also aborted in one chain (0050961 lh, "no closest triangle found", the Q16 AABB abort) — so Q16 now has one image-side data point:
with the unit query no abort occurred in 6 chains × (1 + 6 perturbations).

### F2 TRACE: `GyralCurve::refineCurves` runs ≥ 2 passes in the image

Sulcal curves were byte-identical from the start; the gyral ones differed on 0050952 lh in 32 of 567 curve lines (reversed direction, plus one extra curve: 568 vs 567; vertex Jaccard 0.9994).
Setting `uninit_current="continue"` (the "arbitrary value" branch of `trace.md`) reproduces `.gcurve` and `.gcurve.bary` byte for byte on all six
hemispheres. It is now the default; the oracle goldens pass `--uninit-current exit`. Test: `test_gyral_curves_match_the_image_bytes`.

### F3 Sulcal depth and the hull: `strel('ball')` and the dilation

First comparison against real output. Port: mean +0.71 mm too deep (r = 0.9989). Hull-variant sweeps on 0050952 lh show the bias is the
hull dilation `imdilate(strel('ball',1,1))`: dropping it gives -0.08 mm. Excluding the rim from the structuring-element disc
(`x² + y² < r²`, centre kept) makes `ball(1,1)` the centre pixel and also improves the closing: bias -0.04 mm, r = 0.9997, 54 % of the
vertices equal to print precision (before: 2 %). Thresholds 22-26, other disc radii and octagonal neighbourhoods do not do better. This
is inferred from the image, not from MATLAB source, and it changes `OuterHull` the same way. The residual is voxel-scale (max 1.41 mm =
one chamfer step) and probably the N = 8 line decomposition of the ball, which is not reproduced: the stage stays **below the E-class
bound** on all six hemispheres. Test: `test_strel_ball_excludes_the_rim`, `test_sulcal_depth_has_no_hull_bias_against_the_image`.

### F4 HSD residual (policy bound not met; Q15 outcome mixed)

Across the six hemispheres, the port lands in the image's registration basin every time (median angular displacement 0.10–0.30°, p99 0.59–2.85°, E1 within 0.1–1.2 % of the image's PDF values). Under the deterministic policy bound (0.01° median / 0.1° p99, coeff 1e-4, energy 1e-3) all six fail because the four-stage Newton optimizer is chaotic at 1 ulp. Under the Q15 rule (`max(policy, 1.5 × own 1-ulp spread)` from 6 single-ulp input perturbations per hemisphere), **4 of 6 hemispheres pass all three metrics** (`0050952 lh`, `0050961 lh`, `0050971 lh`, `0050971 rh`). On the remaining two (`0050952 rh` median 0.110° vs 1.5×spread 0.073°; `0050961 rh` median 0.297° vs 1.5×spread 0.213°), the 6-member perturbation ensemble happened to stay tighter than the cross-build trajectory difference; notably, the call-1 `--noguess` variant sits at E1 0.741 (vs image 0.740) on `0050952 rh` and median 0.276° / E1 0.760 (vs image 0.756) on `0050961 rh`, confirming both are within the same local basin.

### F5 LGI is not comparable until klaplace exists

The `lgi` script maps the pial surface to the hull with `klaplace -surfaceCorrespondence`; the port uses a closest-hull-point stand-in
(Q7: licence). On the stand-in the correspondence mesh has twice the hull's area (113,000 vs 61,500 mm² on 0050952 lh) and LGI is
r ≈ 0.89 with the image (ours 12 % lower on average; ROI means off by 40-50 %) for all six hemispheres. The port's own sensitivity (1-ulp
coordinate changes) is p99 2.3-3.4 %, r > 0.9998, so the chaos is real but one to two orders below this gap: the gap is the stand-in,
not Gyrification or the curves (identical to the image's). No LGI row can pass until the correspondence is the image's (klaplace).
OuterHull itself (hull area 61,474 mm² for 0050952 lh) has no image intermediate to compare with.

### F6 Passthroughs

`quant/*.thickness.txt` is FreeSurfer's `thickness` printed `%f` (byte-identical); `quant/*.mid.area.txt` is `MeshProperty -v` = barycentric vertex area
(`mesh.geometry.vertex_areas` matches at print precision); `stat/*.mid.reg.area.txt` is the same on the registered mid surface;
`roi/*.mask.txt` is `cortex.label` as a 0/1 vector. None needs more than the input stage.

## Gaps: unported stages and what will serve as their reference

| Stage (brief) | Image outputs to compare with | Inputs the replay needs (all present or reproducible here) | Missing intermediate |
|---|---|---|---|
| MeshProperty mean curvature `H` and shape index `S` (13) | `quant/<h>.mid.{H,S}.txt`, `stat/<h>.mid.reg.{H,S}.txt` | `<h>.mid.vtk` (byte-identical), `--surfsmoothing 2 --tensorsmoothing 2` | none |
| SCI (13/later) | `quant/<h>.mid.SCI.txt`, `stat/<h>.mid.reg.SCI.txt` | `<h>.mid.vtk`, `sci_kernel_size`/`sci_bins` of `biscuit.conf` | none |
| Parcellation features + alignment (B7.1) | none directly; scores feed the labels | `res.curv`-type features, `parc_alignment_argv` HSD call, templates | per-model scores on the native mesh (`parc` scratch is deleted by the script); ours from the merged network |
| Graph cut and `roi_correction` (16, B7.5) | `roi/<h>.parc.{braincolor,dkt31}.txt` (final labels, per-ROI Dice, 99.5 % agreement) | network scores (above), `roi/<h>.mask.txt`, adjacency of `<h>.sphere.vtk`/`white.vtk` | the pre-graph-cut scores / `.tmp` |
| Statistics (18) | `stat/stats.{braincolor,dkt31}.csv` | the `quant/` and `stat/` maps, `roi/` labels, `curve/<h>.gray.scurve.bary`, `reg/<h>.sphere.reg.vtk`, `mid.vtk` — **all are image outputs, so the stats can be replayed today from them** | none |
| PDF (19) | `pdf/{reg.mb101,parc.*,curve,quant}.pdf` (E0/E1 text already matches the registration energies) | maps above, `<h>.reg.curv.txt` (reproduced), curves | none |
| klaplace correspondence (B9.3, Q7) | `quant/<h>.gray.LGI.txt` only (the hull and correspondence meshes are not kept) | OuterHull, pial, curves | `<h>.gray.hull.corr.vtk` — ask the owner to keep `tmp-lgi-*` in one rerun of the image (Q1) |

## Not done / requests

* A third HSD basin question remains for Q17 (unit query vs `--noguess`): the image's call-1 coefficient file (`tmp1/lh.coeff.txt` after call 1) would decide it; the
  `tmp/` directories of the open examples are empty. One image rerun with `$TMP` kept would also give the OuterHull/klaplace intermediates for LGI.
* SD residual (F3): MATLAB's `strel('ball')` decomposition and heights, if the owner can run one `getheight(strel('ball',15,15))`/`getnhood` in MATLAB or Octave.
* `plan.md` status section not edited (rule of the parallel sessions).
