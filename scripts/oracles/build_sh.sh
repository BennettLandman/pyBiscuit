#!/bin/sh
# Build the SH oracle harness from reference MeshLib sources (no source patches needed).
set -e
ML=${ML:-$HOME/dev/biscuit/reference/repos/MeshLib}
OUT=${1:-$HOME/dev/biscuit/oracles/meshlib-sh}
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
/usr/bin/clang++ -O2 -std=c++14 -include cstddef -I"$ML/include/Util" "$HERE/harness_sh.cpp" \
  "$ML/src/Util/Geom.cpp" "$ML/src/Util/SphericalHarmonics.cpp" -o "$OUT/harness_sh"
echo "$OUT/harness_sh"
