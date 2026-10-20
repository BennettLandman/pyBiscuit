#!/bin/sh
# Build the upstream HSD oracle (f169595, v1.3.9) -> $ORACLES/hsd/bin/HSD
# Flags mirror HSD/CMakeLists.txt + wrapper/CMakeLists.txt: C++11, _USE_SYSV, HSD_VERSION; CUDA off, OpenMP off.
# BLAS/LAPACK: Accelerate via build-only shim headers lapacke.h/cblas.h (patches/hsd-shim). No source edits.
set -e
. "$(dirname "$0")/_common_o1b.sh"
CLI=$(fetch_cli11 1.9.1)
use_meshlib
SHIM=$REPO_ROOT/scripts/oracles/patches/hsd-shim
OUT=$ORACLES/hsd/bin; mkdir -p "$OUT"
S=$REF/HSD
$CXX $CXXFLAGS_COMMON -I"$SHIM" -I"$CLI" -I"$S/src" $ML_INC -D_USE_SYSV '-DHSD_VERSION="v1.3.9"' \
  "$S/src/HSD.cpp" "$S/wrapper/main.cpp" "$LIB" -framework Accelerate -o "$OUT/HSD"
echo "$OUT/HSD"
