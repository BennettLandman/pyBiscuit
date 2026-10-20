# Agent brief 07 — Local gyrification index: OuterHull + Gyrification (B9.2, B9.4)

**Written 2026-10-20.** Runs in parallel with briefs 04–06 after brief 03
merges. Read [AGENTS.md](../../AGENTS.md), [plan.md](../../plan.md), and the
[parallel rules](README.md#parallel-sessions).

## Context

- LGI pipeline (from the image's `cmorph/bin/lgi` tcsh script):
  `OuterHull` → `klaplace` (hull↔surface correspondence, `-dims 128`) →
  `CurveExtraction` (it copies existing `.scurve.bary`/`.gcurve.bary`) →
  `Gyrification -i -o --outer -s -g -m <kernel> --speed <aniso> --refHullArea <ref> --nThreads`.
  BISCUIT defaults: kernel 316, anisotropy 0.2, ref area 77100.
- The image Gyrification reports **v1.2.2**, the same as upstream `b9fa760`.
  The oracle is at `~/dev/biscuit/oracles/gyrification/`; rebuild it with
  `-ffp-contract=off`.
- `OuterHull` is a compiled MATLAB binary in the image; its source is
  `LocalGyrificationIndex/matlab/OuterHull.m` (+ `VOXELISE/`, `ICP_finite/`).
  There is no oracle without MATLAB. If GNU Octave is installable locally,
  use it as the oracle (record its version); otherwise validate against a
  documented re-derivation and mark the row provisional.
- **klaplace is out of scope** (no LICENSE file; Q7 pending). Gyrification
  needs `--outer` (the hull with correspondence). For tests, produce the
  correspondence with the **oracle klaplace** if one can be built; otherwise
  build a fixture where the correspondence is known analytically (e.g. a
  folded sphere with its exact convex hull, matched by radial projection).
- Curves come from TRACE (brief 04, running in parallel). Use oracle
  CurveExtraction outputs as Gyrification inputs here; do not import brief
  04's code until it has merged.

## Scope

| Row | Deliverable |
|---|---|
| B9.2 | `pybiscuit.quant.hull`: OuterHull port (voxelisation, closing, isosurface, ICP_finite alignment, smoothing as in the `.m` files) |
| B9.4 | `pybiscuit.quant.lgi`: Gyrification port (anisotropic Hamilton–Jacobi wavefront kernel on the surface, area-adaptive kernel, hull-area ratio, `--intv`/`-t` intermediate outputs, all flags) and the `lgi` script orchestration minus klaplace; `pybiscuit lgi` subcommand |

## Acceptance

- Gyrification vs oracle on fixtures with identical inputs (surface, outer,
  curves): per-vertex LGI rel ≤ 1e-5, ties explained (brief 03 helpers).
  Kernel maps from `--geodesicMap`: rel ≤ 1e-5.
- OuterHull vs Octave (if available): symmetric surface distance p99 ≤ 0.1 mm
  on fixtures. Otherwise mark the row "provisional (no oracle)".
- Analytic diagnostic: LGI ≈ 1 on a convex sphere; it increases monotonically
  with fold amplitude on `folded_sphere`.
