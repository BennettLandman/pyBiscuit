#!/bin/bash
# Build the upstream HSD oracle (f169595, v1.3.9) with -ffp-contract=off (no fused multiply-add),
# i.e. the float32/float64 arithmetic of the x86-64 reference image, plus an internals harness.
# Companion of build_hsd.sh (which keeps the arm64 default of fused multiply-adds).
#   $ORACLES/hsd-nofma/bin/HSD          same CLI as upstream (wrapper/main.cpp)
#   $ORACLES/hsd-nofma/bin/hsd_harness  scripts/oracles/harness_hsd.cpp (dumps private state)
# Sources: unmodified HSD + MeshLib (dcef807) compiled together; build-only changes: -include cstddef,
# the lapacke.h/cblas.h shim headers (patches/hsd-shim: Accelerate instead of OpenBLAS), -D_USE_SYSV.
# BLAS/LAPACK is still Accelerate (dsysv_, cblas_dgemm, cblas_daxpy), so last-bit differences vs OpenBLAS remain.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT="${BISCUIT_ROOT:-$HOME/dev/biscuit}"
ORA="${ORACLES:-$ROOT/oracles}"
ML="$ROOT/reference/repos/MeshLib"
HSD="$ROOT/reference/repos/HSD"
CLI="$ORA/_deps/cli11-1.9.1"
OUT="$ORA/hsd-nofma"
[ "$(git -C "$ML" rev-parse --short HEAD)" = dcef807 ] || { echo "MeshLib pin mismatch" >&2; exit 1; }
[ "$(git -C "$HSD" rev-parse --short HEAD)" = f169595 ] || { echo "HSD pin mismatch" >&2; exit 1; }
[ -f "$CLI/CLI11.hpp" ] || { echo "missing $CLI/CLI11.hpp (run build_hsd.sh first)" >&2; exit 1; }
CXX=${CXX:-/usr/bin/clang++}
export SDKROOT=${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}
SHIM="$HERE/patches/hsd-shim"
FLAGS="-std=c++11 -O2 -DNDEBUG -ffp-contract=off -w -include cstddef"
INC="-I$SHIM -I$CLI -I$HSD/src -I$HSD/wrapper -I$ML/include -I$ML/include/Util"
mkdir -p "$OUT/bin" "$OUT/obj"
{
  echo "# $($CXX --version | head -1)"
  set -x
  for f in Mesh MeshIO MNIObjIO VtkIO; do $CXX $FLAGS $INC -c "$ML/src/$f.cpp" -o "$OUT/obj/$f.o"; done
  for f in AABB_Sphere AABB Geom SphericalHarmonics SurfaceUtil; do $CXX $FLAGS $INC -c "$ML/src/Util/$f.cpp" -o "$OUT/obj/$f.o"; done
  $CXX $FLAGS $INC -D_USE_SYSV '-DHSD_VERSION="v1.3.9"' -c "$HSD/src/HSD.cpp" -o "$OUT/obj/HSD.o"
  $CXX $FLAGS $INC -D_USE_SYSV '-DHSD_VERSION="v1.3.9"' -c "$HSD/wrapper/main.cpp" -o "$OUT/obj/main.o"
  $CXX $FLAGS $INC -D_USE_SYSV '-DHSD_VERSION="v1.3.9"' -c "$HERE/harness_hsd.cpp" -o "$OUT/obj/harness.o"
  LIBS="$OUT/obj/Mesh.o $OUT/obj/MeshIO.o $OUT/obj/MNIObjIO.o $OUT/obj/VtkIO.o $OUT/obj/AABB_Sphere.o $OUT/obj/AABB.o $OUT/obj/Geom.o $OUT/obj/SphericalHarmonics.o $OUT/obj/SurfaceUtil.o"
  $CXX $FLAGS -o "$OUT/bin/HSD" "$OUT/obj/HSD.o" "$OUT/obj/main.o" $LIBS -framework Accelerate
  $CXX $FLAGS -o "$OUT/bin/hsd_harness" "$OUT/obj/HSD.o" "$OUT/obj/harness.o" $LIBS -framework Accelerate
} > "$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
echo "OK: $OUT/bin/HSD $OUT/bin/hsd_harness"
