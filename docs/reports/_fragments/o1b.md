# O1b fragment: HSD / CurveExtraction / Gyrification oracles

All three built with plain `/usr/bin/clang++` (Apple clang 21.0.0, arm64), `-std=c++11 -O2 -DNDEBUG -include cstddef`,
no OpenMP/CUDA, linked against MeshLib dcef807 (O1a's `oracles/meshlib/lib/libMesh.a`; identical output verified
against my own throwaway build of the same sources). Scripts `scripts/oracles/build_{hsd,curve_extraction,gyrification}.sh`.

| tool | commit | version string | status | smoke time |
|---|---|---|---|---|
| HSD | f169595 | v1.3.9 | built | ~5 s (2 subj, icosphere 5, d=8, ico 4, maxIter 3) |
| CurveExtraction | ded932d | v1.1.2 | built | ~2 s (icosphere 5) |
| Gyrification | b9fa760 | v1.2.2 | built | ~1.6 s (icosphere 5) |

Patches (build-only, no numeric source changes; no `.patch` files, header shims instead):
- `patches/hsd-shim/{lapacke.h,cblas.h}` -> Accelerate (`ACCELERATE_NEW_LAPACK`). HSD uses only `dsysv_`, `cblas_dgemm`, `cblas_daxpy`.
  Numerics come from Accelerate rather than OpenBLAS: expect last-bit/ulp differences vs the Linux image.
- `patches/curve-gyr-omp-shim/omp.h`: serial stub (unguarded `#include <omp.h>`).
- Surprise: Curve/Gyrification need `-D_USE_OPENMP` even with the stub; without it nThreads=0 and CurveExtraction segfaults
  (zero-length per-thread array). Upstream bug in non-OpenMP builds.
- Surprise: CLI11 is `file(DOWNLOAD)`ed by CMake; I fetched v1.9.1 (HSD) and v1.9.0 from the GitHub release URL (sha256 in
  `oracles/_deps/cli11-*`: 1.9.1 `52a3bc82...`, 1.9.0 `6f0a1d88...`).
- Version note: master reports HSD v1.3.9 / TRACE v1.1.2 / LGI v1.2.2; the image-documented CBT v1.3 / cmorph v1.7 could
  not be checked here (no image); no visible behavioural discrepancy in --help, but versions are not equal strings.

Smoke commands: see `scripts/oracles/README.md` (## O1b tools). HSD: two icosphere(5) subjects, s2 rotated 8 deg about z with the
property being the same smooth function of the unrotated coordinates; after `-d 8 --icosahedron 4 --maxIter 3` the max
corresponding-vertex mismatch drops 7.0e-2 -> 1.6e-3; deterministic across runs. CurveExtraction defaults (-0.05/0.03 seeds) are tuned for
real cortex in mm: a synthetic bumpy sphere needs `--sulcalSeed -0.01 --gyralSeed 0.01`. Gyrification output name is
`<o>.lgi.map.<adjusted area>.txt` (area baked into the file name).
No goldens committed (smoke only).
