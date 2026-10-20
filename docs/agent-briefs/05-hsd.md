# Agent brief 05 — HSD spherical registration (B6)

**Written 2026-10-20.** Runs in parallel with briefs 04, 06, 07 after brief 03
merges. Read [AGENTS.md](../../AGENTS.md), [plan.md](../../plan.md),
[parallel rules](README.md#parallel-sessions), and
`docs/ports/spherical-harmonics.md`.

## Context

- The image HSD reports **v1.3.9**, the same as upstream `f169595`. Port from
  that commit.
- The SH basis is already ported (`MeshLib SphericalHarmonics::basis`, used
  at HSD.cpp:365, 826, 1163, 3096). Remaining: everything else in
  `src/HSD.cpp` (3.1k lines) and `src/bobyqa.h` (2.9k lines).
- The oracle is at `~/dev/biscuit/oracles/hsd/` and uses Accelerate LAPACK
  (`dsysv_`, `cblas_dgemm`, `cblas_daxpy`). Rebuild it with `-ffp-contract=off`
  and record both builds. Expect ulp-level differences from OpenBLAS in the image.
- How BISCUIT calls HSD (template mean/var files, degree, icosahedron schedule,
  `--resample`, weights) is hidden in the orchestrator. Port **every** CLI flag
  (`-s -p -o -l -c --surface --weightMap -d --icosahedron --weight --idprior
  --maxIter --locationWeight --icomesh --gres --noguess --resample
  --writecoeff --fixedSubjects --tmpVar --nThreads`, plus the `*Dir` and
  `--filter` batch forms). Work out from the source how the template's
  `*.mean.txt`/`*.var.txt` would be supplied (likely as a fixed subject with
  `--tmpVar`/`--fixedSubjects`), and record that as a hypothesis.

## Scope

| Row | Deliverable |
|---|---|
| B6.1 | `docs/ports/hsd.md`: function map, data structures, precision table (float vs double per array), and OpenMP regions with their reduction order |
| B6.3 | Feature resampling on icosahedra (`--icosahedron`, `--icomesh`, `--resample`), using brief 02's AABB_Sphere |
| B6.4 | Energy (feature term vs mean/variance; location/landmark terms; rigidity/distortion regulariser), gradient, and 2nd-order approximation |
| B6.5 | Optimiser: Levenberg–Marquardt with `dsysv` (SciPy `scipy.linalg.solve(assume_a='sym')`), the BOBYQA port (keep the upstream copyright header; check the `bobyqa.h` licence text first), the initial guess, and the exponential-map update |
| B6.6 | Numba `prange` in place of OpenMP; fixed-order reductions |
| — | `pybiscuit hsd` subcommand with the same flags and output formats (sphere `.vtk`, `--writecoeff` text) |

## Acceptance (E-class vs oracle; provisional)

- Unit level: energy, gradient, and Hessian at fixed coefficients: rel ≤ 1e-8
  (double paths) vs an oracle harness dump.
- Single iteration from identical state: coefficient max abs diff ≤ 1e-6.
- Full runs on fixtures (2–3 synthetic subjects, `folded_sphere` with shifted
  features; degree 8–15; icosahedron 4–5): final energy within 0.1 %, and
  per-vertex angular difference median ≤ 0.01°, p99 ≤ 0.1°.
- Template-style run (fixture subject vs a fixed mean/var "template" built
  from fixtures): same criteria.
- Performance: report wall time per iteration vs the single-thread oracle
  (target ≤ 2×; not gating in this brief).

## Notes

- HSD is the largest single port. If the full scope does not fit in one
  session, finish B6.1, B6.3, and B6.4 with acceptance, and hand back a
  precise plan for B6.5–B6.6.
- CUDA paths (`src/cuda/*`) are out of scope (Q10).
