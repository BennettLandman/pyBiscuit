# Agent brief 02 — Upstream oracles, mesh geometry, spherical remeshing

**Written 2026-10-20.** Read [AGENTS.md](../../AGENTS.md), [plan.md](../../plan.md),
and the [foundations report](../reports/foundations-2026-10-20.md) first.
This brief covers one working session.

## Context

- Brief 01 is done (PR #1): I/O, topology, icospheres, MeshConv operators.
  Branch from `main` once PR #1 is merged. If it is not merged yet, branch
  from `b0-b2-b4-foundations` and say so in your report.
- Still unavailable: the BISCUIT MATLAB source (requested from Ilwoo Lyu) and
  a Linux host for the image (Q1). Do **not** work on orchestrator-dependent
  rows (B1, B5, B7.1, B7.4–B7.5, B9.5, B10). If the MATLAB source arrives,
  stop and report.
- Without image goldens, this session builds **upstream oracles**: the MIT
  C++ tools compiled locally from `reference/repos/`. These are provisional
  references. Image goldens (B1.4) override them later, and any oracle-vs-image
  difference becomes its own finding.

## Scope

| Row | Deliverable |
|---|---|
| H1 | Brief-01 housekeeping (below) |
| O1 | Upstream oracle builds + golden-generation scripts (new; record under B1.4 as "oracle, provisional") |
| B2.4 | MeshLib geometry port: normals, areas, smoothing, curvature tensor (`Mesh.cpp`, `Util/SurfaceUtil.cpp`, `Util/Geom.cpp` as used) |
| B2.5 | `AABB_Sphere` point location + barycentric interpolation |
| B6.2* | MeshLib `Util/SphericalHarmonics.cpp` basis (needed by SphericalRemesh `--deform/-c`; check whether HSD uses the same basis) |
| B4.2 | SphericalRemesh port (all flags) |

Out of scope: geodesics (B2.6, which goes with TRACE in brief 03), HSD optimisation,
MeshProperty black-box work (B3), and any FreeSurfer-replacement decision (Q3).

## H1 — housekeeping from brief 01

1. `pybiscuit.io.read_asc`: remove the `tkRAS` default. `frame` becomes a
   required keyword (`"RAS"` or `"tkRAS"`). Update callers and tests. MaCRUISE
   `.asc` files are scanner RAS (foundations report).
2. `LICENSES/UGSCNN-MeshConv.txt` is a placeholder. Fetch the verbatim LICENSE
   from `https://github.com/maxjiang93/ugscnn` (confirm it is MIT; record the
   commit). `src/pybiscuit/sphere/operators.py` follows meshcnn, so treat it as
   derived and add the attribution header. If the license is not MIT, stop and
   report.
3. B4.1 is recorded as Partial because the shipped spheres are 4-decimal
   quantised. Do not change its status; the owner decides whether to amend the
   tolerance.

## O1 — upstream oracles (macOS arm64)

- Build into `~/dev/biscuit/oracles/<tool>/` (outside git). Use a throwaway
  venv with `pip install cmake` if CMake is easiest; otherwise call clang++
  directly. LAPACK comes from Accelerate (`-framework Accelerate`). CUDA off;
  OpenMP off unless trivial. Build MeshLib, SphericalRemesh, HSD,
  CurveExtraction, and Gyrification (LocalGyrificationIndex `src/`) at the
  commits pinned in [reference-artifacts.md](../reference-artifacts.md).
- Allowed: build-only patches (include paths, missing headers, Accelerate
  symbol names). **Not allowed:** any change to numeric behavior. Commit
  build scripts and patches to `scripts/oracles/`, with one README listing
  compiler version, flags, commits, and every patch.
- Smoke test: run each tool's `--help`, and SphericalRemesh and HSD once on
  committed fixtures (`icosphere(5)` with a synthetic property).
- Golden helper: `scripts/oracles/make_golden.py <tool> <case>` writes small
  outputs to `tests/data/golden/<tool>/<case>/` with `command.json` (argv,
  commit, compiler, `"oracle": "upstream-macos-arm64"`). Keep each case well
  under 1 MB; generate large inputs from fixtures at test time.
- Tests that call the oracle binaries must skip when `oracles/` is absent (CI).
  Committed goldens must not need the binaries.
- If a tool cannot be built without changing numerics, record it as Blocked
  with the reason and move on.

## B2.4 — geometry (port of MeshLib, E-class vs oracle)

- Read `MeshLib/src/Mesh.cpp` (normals, areas) and `src/Util/SurfaceUtil.cpp`
  in full. Port `smoothing` (all overloads: iterations, sigma, data
  smoothing), `tensor`, `smoothingTensor`, `principalCurvature`, `curvature`,
  and `curvature_quad`. Also port `SurfaceUtil::sphere(mesh, type, max_iter)`:
  it is MeshLib's spherical mapping and a candidate for B5.7. Port it only;
  do not adopt it as the replacement (that is Q3).
- Precision: MeshLib stores vertices as `float`. Keep float32 wherever
  rounding can change a decision or an iteration; document each choice in
  `docs/ports/meshlib-geometry.md`.
- Oracle driver: a ~50-line C++ harness linked against the oracle MeshLib
  that dumps these quantities for a mesh. Keep it in `scripts/oracles/`.
- **Accept:** on fixtures plus `reg/mb101/lh.white.vtk`: per-vertex relative
  difference ≤ 1e-5 (float32 paths) or ≤ 1e-10 (double paths); principal
  directions agree up to sign. These curvature functions are also the
  starting point for MeshProperty (B3); note in the port note which
  MeshProperty flags they seem to correspond to, without claiming parity.

## B2.5 — AABB_Sphere + barycentric

- Port `AABB_Sphere` (`closestFace` overloads including the `eps1/eps2`
  candidate logic, `update`, tree construction) and the barycentric
  coefficient computation it returns. The tree layout is internal and need not
  match, but the results must.
- **Accept:** for 10⁵ random unit-sphere points (seeded) on `icosphere(5..7)`
  and on a deformed sphere fixture: identical face index except exact ties
  (count and report them); barycentric coefficients ≤ 1e-6 abs. Must run in
  ≤ 2 s for 163,842 queries on ico7 (Numba allowed).

## B6.2* / B4.2 — spherical harmonics and SphericalRemesh

- Port `MeshLib/src/Util/SphericalHarmonics.cpp`. Compare against the
  oracle's SH values at fixture points up to degree 20. Check whether
  `HSD/src/HSD.cpp` uses this basis or its own, and record that in
  `docs/ports/spherical-harmonics.md`. It sets the scope of B6.2.
- Port `SphericalRemesh/src/SphericalRemeshing.cpp` with every CLI flag in
  `wrapper/PARSE_ARGS.h` (`-s -i -c -r -p --color -o --outputProperty --deform
  --bary --ftype --nneighbor --deg --deg0 --keepColor --quiet`). Expose it as
  `pybiscuit.sphere.remesh` and as a `pybiscuit spherical-remesh` subcommand
  with the same flag names.
- The parcellation path calls
  `SphericalRemesh -p <list> -r <native sphere> -s <ico sphere> --outputProperty <prefix> --quiet`
  (`reference/ctf_plain/SurfPostproc/common/parc/test.py`). Make that a named
  test case.
- **Accept:** E-class vs oracle on fixtures (property resampling, nearest
  neighbour, barycentric output, deformation with SH coefficients): per-vertex
  ≤ 1e-5 relative; output files have identical text formatting.

## Evidence and hand-back

- Branch `b2-b4-geometry-remesh`; PR to `main`; commits reference row IDs.
- Update plan.md row Status cells (B2.4, B2.5, B4.2; add B6.2 progress; add
  "O1 oracles" under B1.4's notes) with tests, commands, counts, and commit.
- Report: `docs/reports/geometry-remesh-YYYY-MM-DD.md`, with measured
  tolerances, tie counts, timings, oracle build notes, and surprises.
- Port notes: `docs/ports/{oracles,meshlib-geometry,aabb-sphere,spherical-harmonics,spherical-remesh}.md`.
- Final message: rows done or not done with reasons, oracle build status per
  tool, surprises, and proposed next rows. Brief 03 will likely be geodesics +
  TRACE, or HSD.

## Stop conditions

Stop and report if: an oracle needs numeric source changes to build; upstream
`master` visibly differs from documented image behavior (cmorph v1.7,
June 2021); a new runtime dependency seems necessary; the UGSCNN license is
not MIT; or the MATLAB source arrives.
