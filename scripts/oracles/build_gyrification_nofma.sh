#!/bin/bash
# Build upstream Gyrification (LocalGyrificationIndex b9fa760, v1.2.2) with -ffp-contract=off, compiling the
# unmodified MeshLib (dcef807) + Gyrification sources directly: the float32/double behaviour of the x86-64 image.
# Companion of build_gyrification.sh (arm64 default FMA).  Build-only changes: -include cstddef, the serial omp.h
# stub (patches/curve-gyr-omp-shim) and -D_USE_OPENMP (see docs/ports/oracles.md).
# libm: scripts/oracles/patches/acosf-fdlibm/acosf.c replaces the host `acosf` (fdlibm algorithm; provisional pin)
# because the LGI kernels are sensitive to 1 ulp of MeshLib's float32 normals (see docs/reports/lgi-hull-2026-10-20.md).
# Output: $ORACLES/gyrification-nofma/bin/Gyrification
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${BISCUIT_ROOT:-$HOME/dev/biscuit}"
ORA="${ORACLES:-$ROOT/oracles}"
ML="$ROOT/reference/repos/MeshLib"
S="$ROOT/reference/repos/LocalGyrificationIndex"
OUT="$ORA/gyrification-nofma"
[ "$(git -C "$ML" rev-parse --short HEAD)" = dcef807 ] || { echo "MeshLib pin mismatch" >&2; exit 1; }
[ "$(git -C "$S" rev-parse --short HEAD)" = b9fa760 ] || { echo "LocalGyrificationIndex pin mismatch" >&2; exit 1; }
CLI=$ORA/_deps/cli11-1.9.0
[ -s "$CLI/CLI11.hpp" ] || { echo "CLI11 1.9.0 missing: run build_gyrification.sh first" >&2; exit 1; }
export SDKROOT=${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}
CXX=${CXX:-/usr/bin/clang++}
SHIM="$HERE/patches/curve-gyr-omp-shim"
mkdir -p "$OUT/bin"
INC="-I$ML/include -I$ML/include/Geodesic -I$ML/include/Geodesic/gw -I$ML/include/Geodesic/gw/gw_core -I$ML/include/Geodesic/gw/gw_geodesic -I$ML/include/Geodesic/gw/gw_maths -I$ML/include/Util"
SRCS=$(grep -o 'src/[A-Za-z0-9_/]*\.cpp' "$ML/CMakeLists.txt" | sed "s#^#$ML/#")
{
  echo "# $($CXX --version | head -1)"
  set -x
  $CXX -std=c++11 -O2 -DNDEBUG -ffp-contract=off -w -include cstddef -D_USE_OPENMP -I"$SHIM" -I"$CLI" -I"$S/src" $INC \
    '-DLGI_VERSION="v1.2.2"' "$S/src/Gyrification.cpp" "$S/wrapper/main.cpp" $SRCS \
    "$HERE/patches/acosf-fdlibm/acosf.c" -o "$OUT/bin/Gyrification"
} > "$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
echo "OK: $OUT/bin/Gyrification"
