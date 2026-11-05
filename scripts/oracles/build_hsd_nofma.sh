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
# HSD_PATCH_ICOMESH=1: also fix the dangling pointer in the --icomesh branch of HSD::init (HSD.cpp:316,
# `const float *p = V.unit().fv();` reads a destroyed temporary; clang -O2 reuses its stack slot, so every
# icomesh sample becomes garbage and the run dies).  The patched build goes to hsd-icomesh-nofma and is
# otherwise identical; the image's gcc build is assumed to keep the value (queue: Q1).
# HSD_PATCH_LANDMARKS=1: also apply patches/hsd-landmarks/PARSE_ARGS.h.patch, which uncomments the options
# upstream disabled (-l/--landmark, --landmarkDir, --surface, --surfaceDir, --weightMap, --locationWeight) so the
# landmark term, location features and BOBYQA pole search (src/bobyqa.h, compiled and run as a black box) can be
# reached.  Output: hsd-lm-nofma (no other change; the icomesh fix is not applied).
if [ "${HSD_PATCH_ICOMESH:-0}" = 1 ]; then OUT="$ORA/hsd-icomesh-nofma"
elif [ "${HSD_PATCH_LANDMARKS:-0}" = 1 ]; then OUT="$ORA/hsd-lm-nofma"
else OUT="$ORA/hsd-nofma"; fi
[ "$(git -C "$ML" rev-parse --short HEAD)" = dcef807 ] || { echo "MeshLib pin mismatch" >&2; exit 1; }
[ "$(git -C "$HSD" rev-parse --short HEAD)" = f169595 ] || { echo "HSD pin mismatch" >&2; exit 1; }
[ -f "$CLI/CLI11.hpp" ] || { echo "missing $CLI/CLI11.hpp (run build_hsd.sh first)" >&2; exit 1; }
CXX=${CXX:-/usr/bin/clang++}
export SDKROOT=${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}
SHIM="$HERE/patches/hsd-shim"
FLAGS="-std=c++11 -O2 -DNDEBUG -ffp-contract=off -w -include cstddef"
WRAP="$HSD/wrapper"
if [ "${HSD_PATCH_LANDMARKS:-0}" = 1 ]; then
  WRAP="$OUT/wrapper"
  mkdir -p "$WRAP"
  cp "$HSD/wrapper/main.cpp" "$HSD/wrapper/PARSE_ARGS.h" "$WRAP/"
  patch -s -d "$OUT" -p1 < "$HERE/patches/hsd-landmarks/PARSE_ARGS.h.patch" || { echo "landmark patch did not apply" >&2; exit 1; }
fi
INC="-I$SHIM -I$CLI -I$HSD/src -I$WRAP -I$ML/include -I$ML/include/Util"
mkdir -p "$OUT/bin" "$OUT/obj"
HSD_SRC="$HSD/src/HSD.cpp"
if [ "${HSD_PATCH_ICOMESH:-0}" = 1 ]; then
  mkdir -p "$OUT/src"
  HSD_SRC="$OUT/src/HSD.cpp"
  sed '316s|const float \*p = V.unit().fv();|Vector U = V.unit(); const float *p = U.fv();|' "$HSD/src/HSD.cpp" > "$HSD_SRC"
  [ "$(diff "$HSD/src/HSD.cpp" "$HSD_SRC" | grep -c '^>')" = 1 ] || { echo "icomesh patch did not apply" >&2; exit 1; }
fi
{
  echo "# $($CXX --version | head -1)"
  set -x
  for f in Mesh MeshIO MNIObjIO VtkIO; do $CXX $FLAGS $INC -c "$ML/src/$f.cpp" -o "$OUT/obj/$f.o"; done
  for f in AABB_Sphere AABB Geom SphericalHarmonics SurfaceUtil; do $CXX $FLAGS $INC -c "$ML/src/Util/$f.cpp" -o "$OUT/obj/$f.o"; done
  $CXX $FLAGS $INC -D_USE_SYSV '-DHSD_VERSION="v1.3.9"' -c "$HSD_SRC" -o "$OUT/obj/HSD.o"
  $CXX $FLAGS $INC -D_USE_SYSV '-DHSD_VERSION="v1.3.9"' -c "$WRAP/main.cpp" -o "$OUT/obj/main.o"
  $CXX $FLAGS $INC -D_USE_SYSV '-DHSD_VERSION="v1.3.9"' -c "$HERE/harness_hsd.cpp" -o "$OUT/obj/harness.o"
  LIBS="$OUT/obj/Mesh.o $OUT/obj/MeshIO.o $OUT/obj/MNIObjIO.o $OUT/obj/VtkIO.o $OUT/obj/AABB_Sphere.o $OUT/obj/AABB.o $OUT/obj/Geom.o $OUT/obj/SphericalHarmonics.o $OUT/obj/SurfaceUtil.o"
  $CXX $FLAGS -o "$OUT/bin/HSD" "$OUT/obj/HSD.o" "$OUT/obj/main.o" $LIBS -framework Accelerate
  $CXX $FLAGS -o "$OUT/bin/hsd_harness" "$OUT/obj/HSD.o" "$OUT/obj/harness.o" $LIBS -framework Accelerate
} > "$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
echo "OK: $OUT/bin/HSD $OUT/bin/hsd_harness"
