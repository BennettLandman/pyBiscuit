#!/bin/sh
# Build the B2.5 oracle harness against the *unmodified* upstream MeshLib sources.
# usage: build_harness_aabb.sh [OUT_DIR]   (default ~/dev/biscuit/oracles/meshlib/bin)
# Needs only clang++ (no cmake).  -ffp-contract=off: no fused multiply-add, to
# match the x86-64 reference image.  OpenMP/CUDA off.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ML="${MESHLIB_SRC:-$HERE/../../../reference/repos/MeshLib}"
OUT="${1:-$HOME/dev/biscuit/oracles/meshlib/bin}"
mkdir -p "$OUT"
CXX="${CXX:-clang++}"
$CXX -O2 -std=c++11 -ffp-contract=off -w \
  -I"$ML/include" -I"$ML/include/Util" \
  "$HERE/harness_aabb.cpp" \
  "$ML/src/Mesh.cpp" "$ML/src/MeshIO.cpp" "$ML/src/MNIObjIO.cpp" "$ML/src/VtkIO.cpp" \
  "$ML/src/Util/AABB_Sphere.cpp" "$ML/src/Util/Geom.cpp" \
  -o "$OUT/harness_aabb"
echo "built $OUT/harness_aabb"
$CXX --version | head -1
git -C "$ML" rev-parse HEAD 2>/dev/null || true
