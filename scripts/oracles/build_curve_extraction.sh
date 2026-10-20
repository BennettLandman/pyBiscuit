#!/bin/sh
# Build the upstream CurveExtraction oracle (ded932d, v1.1.2) -> $ORACLES/curve_extraction/bin/CurveExtraction
set -e
. "$(dirname "$0")/_common_o1b.sh"
CLI=$(fetch_cli11 1.9.0)
use_meshlib
SHIM=$REPO_ROOT/scripts/oracles/patches/curve-gyr-omp-shim
OUT=$ORACLES/curve_extraction/bin; mkdir -p "$OUT"
S=$REF/CurveExtraction
# _USE_OPENMP only selects the thread-count default (>=1); omp.h is the serial stub in patches/curve-gyr-omp-shim, so the run is single-threaded.
$CXX $CXXFLAGS_COMMON -D_USE_OPENMP -I"$SHIM" -I"$CLI" -I"$S/src" $ML_INC '-DTRACE_VERSION="v1.1.2"' \
  "$S"/src/GyralCurve.cpp "$S"/src/GyralPoint.cpp "$S"/src/SulcalCurve.cpp "$S"/src/SulcalPoint.cpp "$S"/src/Extraction.cpp \
  "$S/wrapper/main.cpp" "$LIB" -o "$OUT/CurveExtraction"
echo "$OUT/CurveExtraction"
