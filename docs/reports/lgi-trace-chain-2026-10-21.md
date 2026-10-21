# Brief 20 report: TRACE wired into the LGI pipeline (B20.1, B20.2) — 2026-10-21

Branch `brief-20-trace-lgi` (from `main` `1e76635`). All validation is **provisional** (macOS arm64 no-FMA oracles,
not the image; Q1). klaplace untouched (licence, Q7).

## What was built

| Piece | Code | Spec |
|---|---|---|
| BISCUIT's four TRACE calls (argv builder, lh→rh, sulcus on `gray` then gyrus on `white`, skip-if-exists) | `src/pybiscuit/trace/pipeline.py` | §8, `curve_extract_pipeline.m:37-47` |
| `lgi` script chain per hemisphere: `OuterHull` → hull correspondence → curves → `Gyrification -m 316.000000 --speed 0.200000 --refHullArea 77100.000000` → `<h>.gray.LGI.txt` | `src/pybiscuit/quant/lgi_pipeline.py` | §9.3 |
| CLI `pybiscuit lgi-pipeline --surf-dir D --out O` (TRACE + LGI; accepts `--trace_r --trace_delta --lgi_kernel[_size] --lgi_aniso[tropy] --lgi_ref[_area]`) | `quant/lgi_pipeline_cli.py`, `cli.py` | — |
| Oracle-chain golden (oracle CurveExtraction ×2 + oracle Gyrification, BISCUIT argv) | `scripts/oracles/make_golden_lgi_chain.py`, `tests/data/golden/lgi-chain/` | — |
| Template comparison script (port chain vs oracle chain) | `scripts/validation/lgi_chain_vs_oracle.py` | — |
| Tests | `tests/test_lgi_chain.py` (14) | — |

`pybiscuit lgi` stays the plain `Gyrification` CLI; the driver is a new subcommand so the original flags are not overloaded.

## Mismatches from the spec (§3.1, §3.4 of the orchestrator-spec report)

| # | Resolution |
|---|---|
| T-1 | `docs/ports/trace.md` hypothesis paragraph replaced by the real call (`--epRad r --nhDist r --lsTh d`, `num2str` formatting). Named test `test_biscuit_trace_argv_is_the_spec_call`; `trace_r`/`trace_delta` now reach the tool (`curve_argv`). |
| T-2 | Default seeds exercised on cortex scale (template, below) and on a new fold fixture (a = 0.25) that yields both sulcal and gyral points at the defaults. |
| T-3 | **Still open.** No hemisphere pial surface exists locally (the sample MaCRUISE `outerSurf` is whole-brain, before hemisphere separation, brief 12). The template run uses `lh.white` as both `gray` and `white`. |
| T-4 | Test with a flattened polar cap (stand-in for the medial patch): chain runs, curves are not masked upstream, LGI finite. Weak by nature (upstream has no mask to test). |
| T-7 | `test_curves_unchanged_by_six_decimal_vtk_rounding`: inputs rounded to `%f` give byte-identical curves. |
| T-5, T-6, T-8 | unchanged (T-6 `GyralCurve` uninitialised `current` needs the image, §15 item 8). |
| L-1 | `test_biscuit_gyrification_argv_has_percent_f_numbers`; port note `lgi.md` updated (driver now exists). |
| L-3 | Hull area reported below; correspondence is a documented stand-in (next section). |
| L-2, L-6 | OK, exercised by the chain. |

## The one non-image step: hull correspondence

`klaplace -surfaceCorrespondence` is not reproduced. `hull_correspondence` moves each pial vertex to the **closest point on the
OuterHull surface** (same faces). Both chains in every comparison share it, so the comparison isolates TRACE + Gyrification.
LGI values are therefore *not* comparable to the image until klaplace goldens exist (Q1/Q7). First attempt (snap to nearest hull
*vertex*) gave 72 % exactly duplicated outer vertices; the closest-point version replaced it (results in the table).

## Results

### Fixtures (in CI): `tests/test_lgi_chain.py`
Fold fixtures with the vertex-wise hull of brief 07 (the test correspondence approach). Ported TRACE with BISCUIT's argv → `.scurve`,
`.scurve.bary`, `.gcurve`, `.gcurve.bary` **byte-identical** to oracle CurveExtraction (default seeds); LGI from the ported curves vs
oracle chain: rel ≤ 1e-5 (policy default for float32). Gyrification adjusted size 82.3 for ref hull area 77100 (hull ratio 0.26).

### Template `reg/mb101/lh.white.vtk` (163,842 vertices), used as pial and white; 4 threads
Metrics: `docs/reports/lgi-trace-chain-2026-10-21/metrics-template-lh-white.json`.

| Quantity | Result |
|---|---|
| Curve files (4) port vs oracle | byte-identical; vertex Jaccard 1.0 (1,955 curve vertices) |
| OuterHull area / surface area / correspondence-mesh area | 51,786 / 52,827 / 56,302 mm² |
| LGI (oracle) median / range | 1.66 / 1.02–8.14 |
| LGI port vs oracle: Pearson r | 0.9999998 (≥ 0.9999 ✔) |
| LGI port vs oracle: rel diff | p99 6.6e-4; max 1.1e-2; 2,832 of 163,842 vertices > 1e-4 (**fails** the E-class 1e-4 per-vertex bound) |
| Timing | TRACE oracle 424 s vs port 53 s; Gyrification oracle 794 s vs port 98 s |

The LGI row is **not a pass** of `validation-tolerances.md` (E: ≤ 1e-4 relative). It is reported with its measured numbers for the
owner (the policy forbids loosening to pass). Context: the curves are identical, so the residual sits in Gyrification at cortex
scale. Sensitivity probe (same inputs): switching the port's `acosf` from fdlibm to the host libm (ulp-level change of the normals)
changes 55,335 vertices by up to 1.8 %, i.e. the kernel-membership cutoff makes LGI chaotic at this scale and the port-vs-oracle
residual (max 1.1 %) is smaller than the effect of a 1-ulp libm difference. With the earlier vertex-snapped hull the numbers were
max 1.3e-2, 3,510 vertices > 1e-4 (`metrics-…-vertexsnap.json`), so it is not caused by the hull. I did not find the remaining source;
candidates: float32 accumulation or ordering inside kernel area sums or tensor smoothing, which the 41k-vertex scale check (3e-7)
does not reach. Proposed next step: bisect with intermediate dumps from the oracle (geodesic map `--geodesicMap`, `--kArea1/2`) on this
template; the image run (Q1) shows whether the image agrees with the oracle or with the port.

## Not done / requests
* Pial-type hemisphere surface with a patch (T-3): after brief 12 merges, run `lgi_chain_vs_oracle.py` on a real separated hemisphere.
* `pipeline/stages.py` (brief 08's file): the `trace` and `quant` stage `blocked` texts are stale (they say TRACE/LGI are not wired or not on main). I did not edit it. The driver functions `curve_extract_pipeline` and `lgi_pipeline` are ready to be called from stages once the surface-prep outputs exist.
* Hull area of a real **pial** surface (L-3) is still unmeasured; the template white gives 51,786 mm².
