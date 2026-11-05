#!/bin/bash
# Build the upstream CurveExtraction (TRACE) oracle and a MeshLib geodesic/slicer harness with
# -ffp-contract=off (no fused multiply-add), i.e. the float32 arithmetic of the x86-64 image
# binary.  Compiles the *unmodified* MeshLib (dcef807) and CurveExtraction (ded932d) sources
# directly; build-only changes: -include cstddef (Geom.h uses NULL), serial omp.h stub, -D_USE_OPENMP
# (see docs/ports/oracles.md).  Companion of build_curve_extraction.sh (arm64 default: FMA fused).
# Output: $ORACLES/curve_extraction-nofma/bin/{CurveExtraction,harness_geodesic}
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${BISCUIT_ROOT:-$HOME/dev/biscuit}"
ORA="${ORACLES:-$ROOT/oracles}"
ML="$ROOT/reference/repos/MeshLib"
CE="$ROOT/reference/repos/CurveExtraction"
OUT="$ORA/curve_extraction-nofma"
[ "$(git -C "$ML" rev-parse --short HEAD)" = dcef807 ] || { echo "MeshLib pin mismatch" >&2; exit 1; }
[ "$(git -C "$CE" rev-parse --short HEAD)" = ded932d ] || { echo "CurveExtraction pin mismatch" >&2; exit 1; }
CLI="$ORA/_deps/cli11-1.9.0"
[ -s "$CLI/CLI11.hpp" ] || { echo "run build_curve_extraction.sh first (CLI11 1.9.0 cache)" >&2; exit 1; }
CXX=${CXX:-/usr/bin/clang++}
export SDKROOT=${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}
SHIM="$HERE/patches/curve-gyr-omp-shim"
INC="-I$ML/include -I$ML/include/Geodesic -I$ML/include/Geodesic/gw -I$ML/include/Geodesic/gw/gw_core -I$ML/include/Geodesic/gw/gw_geodesic -I$ML/include/Geodesic/gw/gw_maths -I$ML/include/Util"
FLAGS="-std=c++11 -O2 -DNDEBUG -ffp-contract=off -w -include cstddef"
mkdir -p "$OUT/bin" "$OUT/obj"
{
  echo "# $($CXX --version | head -1)"; echo "# MeshLib dcef807, CurveExtraction ded932d"; echo "# flags: $FLAGS"
  cd "$OUT/obj"
  i=0
  for f in $(grep -o 'src/[A-Za-z0-9_/]*\.cpp' "$ML/CMakeLists.txt"); do
    o="$(echo "${f%.cpp}" | tr / _).o"
    $CXX $FLAGS $INC -c "$ML/$f" -o "$o" & i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
  wait
  rm -f ../libMesh.a; ar rcs ../libMesh.a *.o; ranlib ../libMesh.a
  $CXX $FLAGS -D_USE_OPENMP -I"$SHIM" -I"$CLI" -I"$CE/src" $INC '-DTRACE_VERSION="v1.1.2"' \
    "$CE"/src/GyralCurve.cpp "$CE"/src/GyralPoint.cpp "$CE"/src/SulcalCurve.cpp "$CE"/src/SulcalPoint.cpp \
    "$CE"/src/Extraction.cpp "$CE/wrapper/main.cpp" ../libMesh.a -o "$OUT/bin/CurveExtraction"
  # harnesses (geodesic distances / paths, slicer contours, gw triangular interpolation)
  for h in harness_geodesic harness_slicer harness_interpolation; do
    [ -f "$HERE/$h.cpp" ] && $CXX $FLAGS $INC "$HERE/$h.cpp" ../libMesh.a -o "$OUT/bin/$h"
  done
} > "$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
echo "OK: $OUT/bin/CurveExtraction"
