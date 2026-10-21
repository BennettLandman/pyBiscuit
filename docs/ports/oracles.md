# Upstream oracles

Provisional oracles (record under B1.4 as "oracle, provisional"): upstream C++ built on the
owner's macOS arm64 workstation, outside git, into `~/dev/biscuit/oracles/<tool>/`. They are
not the acceptance reference (that is the SIF image on Linux x86-64) but pin upstream-source
behaviour for E-class ports. Scripts and patches: `scripts/oracles/`; usage:
`scripts/oracles/README.md`. Golden helper: `scripts/oracles/make_golden.py`.

## Toolchain

- Compiler: Apple clang version 21.0.0 (clang-2100.3.34.2), `/usr/bin/clang++`, arm64 (Darwin).
- No cmake: sources and defines taken from each upstream `CMakeLists.txt`.
- Flags: `-std=c++11 -O2 -DNDEBUG` (CMake would use RelWithDebInfo/Release; `-O2` here, no
  `-ffast-math`, no OpenMP, no CUDA). arm64 clang defaults to FP contraction (FMA fusion) on;
  we did not disable it, so last-bit differences vs x86-64 builds are possible (see fragment).
- Third-party build input: CLI11 v1.9.1 single header (upstream CMake downloads it), from
  `https://github.com/CLIUtils/CLI11/releases/download/v1.9.1/CLI11.hpp`; sha256 is in
  `oracles/spherical-remesh/build.log`.

## MeshLib (O1a)

- Commit `dcef807` (verified with `git rev-parse --short HEAD`; reference/ untouched).
- Sources: exactly those in CMakeLists (`Mesh, MeshIO, MNIObjIO, VtkIO`, `Util/{Geom, AABB,
  AABB_Sphere, Slicer, SphericalHarmonics, SurfaceUtil}`, `Geodesic/*` incl. the `gw` subset
  listed there; other gw files such as `GW_Parameterization` are not in the CMake and not built).
- Output: `oracles/meshlib/{lib/libMesh.a, include/, src/, build.log, READY}`; `include/` and
  `src/` are patched copies of the upstream trees.
- Patches (`scripts/oracles/patches/`):
  1. `meshlib-01-geom-cstddef.patch`: add `#include <cstddef>` to `include/Util/Geom.h`
     (`NULL` undeclared when `SphericalHarmonics.cpp` is compiled first on libc++). Build-only.
- Status: **built**, all targets compile (warnings only: mismatched `new[]`/`delete`, deprecated
  `sprintf`).

## SphericalRemesh (O1a)

- Commit `b677c39`, version string `v1.2.1` (`-DSREMESH_VERSION`), sources
  `src/SphericalRemeshing.cpp` + `wrapper/main.cpp`, linked with `libMesh.a`.
- Patches: none.
- Status: **built**; `--help` OK; smoke on seeded fixtures icosphere(5) -> icosphere(4) and (6),
  see `tests/test_oracle_smoke_remesh.py`. Goldens: `tests/data/golden/spherical-remesh/{down5to4,up3to4}`.

## HSD

- Commit `f169595`, version string `v1.3.9` (`HSD_VERSION`), sources `src/HSD.cpp` + `wrapper/main.cpp`
  (CLI11 v1.9.1), linked with `libMesh.a` and `-framework Accelerate`; defines `_USE_SYSV` (CMake default
  `BLAS_SYSV=ON`); CUDA and OpenMP off. Script: `scripts/oracles/build_hsd.sh`.
- Build-only patches: `patches/hsd-shim/{lapacke.h,cblas.h}` forward both headers to `<Accelerate/Accelerate.h>`
  with `ACCELERATE_NEW_LAPACK` (HSD.cpp uses `dsysv_`, `cblas_dgemm`, `cblas_daxpy` only; the new-LAPACK
  prototypes accept the existing `char*`/`int*` calls). No upstream source edit. Numeric caveat: dsysv/dgemm
  come from Accelerate, not OpenBLAS (upstream's `-lopenblas`), so last-bit differences vs Linux are expected.
- Status: **built**; `--help` OK; smoke run on two icosphere(5) subjects (second rotated 8 deg about z,
  same function of the unrotated coords): `-d 8 --icosahedron 4 --maxIter 3` takes ~5 s, energy ~1e-6,
  max vertex mismatch between registered subjects 1.6e-3 (was 7e-2); output identical with `--nThreads 2`
  (OpenMP off, flag ignored). Test: `tests/test_oracle_smoke_hsd.py`.

## CurveExtraction

- Commit `ded932d`, version string `v1.1.2` (`TRACE_VERSION`), sources `src/{GyralCurve,GyralPoint,SulcalCurve,
  SulcalPoint,Extraction}.cpp` + `wrapper/main.cpp` (CLI11 v1.9.0), linked with `libMesh.a`. Script:
  `scripts/oracles/build_curve_extraction.sh`.
- Build-only: `patches/curve-gyr-omp-shim/omp.h` (serial stub: sources `#include <omp.h>` unguarded and Apple clang
  has no OpenMP) and `-D_USE_OPENMP`. The define is REQUIRED: without it `nThreads` stays 0, `extraction()`
  allocates zero-length per-thread arrays and the tool segfaults (upstream bug when built without OpenMP). With
  the stub it is a single-thread run (identical to `OMP_NUM_THREADS=1`).
- Status: **built**; `--help` OK; smoke on a synthetic bumpy sphere (icosphere(5) radially modulated, ~50 mm):
  default seeds (-0.05/0.03) find no sulcal points, so use `--sulcalSeed -0.01 --gyralSeed 0.01` (9 scurves,
  161 gcurves); ~2 s. Test: `tests/test_oracle_smoke_curve.py`.

## Gyrification

- Source tree: LocalGyrificationIndex `b9fa760`, `src/Gyrification.cpp` + `wrapper/main.cpp` (CLI11 v1.9.0),
  version string `v1.2.2` (`LGI_VERSION`), linked with `libMesh.a`. Script: `scripts/oracles/build_gyrification.sh`.
- Build-only: same omp.h stub and `-D_USE_OPENMP` as CurveExtraction (same thread-count default issue).
- Status: **built**; `--help` OK; smoke: `Gyrification -i bumpy.vtk --outer hull.vtk -g x.gcurve -s x.scurve -o lgi`
  (hull = sphere r=55) writes `lgi.lgi.map.<adjusted area>.txt`, values in [1.0, 1.14], ~1.6 s.
  Test: `tests/test_oracle_smoke_gyr.py` (also needs the CurveExtraction oracle).

## Note: FP contraction (added at hand-back)

Apple clang on arm64 fuses multiply-adds by default. Ports B2.4a/b, B2.5 and B4.2 are bit-identical to MeshLib built with `-ffp-contract=off` (which matches the x86-64 image binary's arithmetic); against a default-flag build, curvature differs by ~1e-5 median and closest-face ties appear. `build_meshlib.sh` now defaults to `-ffp-contract=off`; the existing `oracles/meshlib/lib` predates this change and should be rebuilt before new work uses it. The committed goldens were generated with non-FMA builds (per-case `command.json`).

## Gyrification, non-FMA build (brief 07)

`scripts/oracles/build_gyrification_nofma.sh` compiles the unmodified MeshLib `dcef807` sources and
LocalGyrificationIndex `b9fa760` (`Gyrification.cpp` + `wrapper/main.cpp`, CLI11 v1.9.0) with
`-ffp-contract=off` into `oracles/gyrification-nofma/bin/Gyrification` (build-only: `-include cstddef`,
the serial `omp.h` stub, `-D_USE_OPENMP`). Companion of the FMA build above. The golden generator
`scripts/oracles/make_golden_gyrification.py` also needs `oracles/curve_extraction-nofma/`. Pitfalls found:
MeshLib `VtkIO` reads only **ASCII** VTK (a binary file hangs the tool), and CLI11 rejects a repeated
single-valued option (`-s` twice).
