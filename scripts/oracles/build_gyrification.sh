#!/bin/sh
# Build the upstream Gyrification (LocalGyrificationIndex b9fa760, v1.2.2) oracle -> $ORACLES/gyrification/bin/Gyrification
set -e
. "$(dirname "$0")/_common_o1b.sh"
CLI=$(fetch_cli11 1.9.0)
use_meshlib
SHIM=$REPO_ROOT/scripts/oracles/patches/curve-gyr-omp-shim
OUT=$ORACLES/gyrification/bin; mkdir -p "$OUT"
S=$REF/LocalGyrificationIndex
# _USE_OPENMP only selects the thread-count default (>=1); omp.h is the serial stub in patches/curve-gyr-omp-shim, so the run is single-threaded.
$CXX $CXXFLAGS_COMMON -D_USE_OPENMP -I"$SHIM" -I"$CLI" -I"$S/src" $ML_INC '-DLGI_VERSION="v1.2.2"' \
  "$S/src/Gyrification.cpp" "$S/wrapper/main.cpp" "$LIB" -o "$OUT/Gyrification"
echo "$OUT/Gyrification"
