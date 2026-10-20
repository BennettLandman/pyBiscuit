#!/bin/bash
# Build upstream SphericalRemesh with -ffp-contract=off (no fused multiply-add), i.e. the
# float32 behaviour of the x86-64 reference image, compiling the *unmodified* MeshLib and
# SphericalRemesh sources directly (companion of build_spherical_remesh.sh, which keeps the
# arm64 default of fused multiply-adds).  Only build change: -include cstddef (Geom.h uses NULL).
# Output: $ORACLES/spherical-remesh-nofma/bin/SphericalRemesh
set -euo pipefail
ROOT="${BISCUIT_ROOT:-$HOME/dev/biscuit}"
ORA="${ORACLES:-$ROOT/oracles}"
ML="$ROOT/reference/repos/MeshLib"
SR="$ROOT/reference/repos/SphericalRemesh"
OUT="$ORA/spherical-remesh-nofma"
[ "$(git -C "$ML" rev-parse --short HEAD)" = dcef807 ] || { echo "MeshLib pin mismatch" >&2; exit 1; }
[ "$(git -C "$SR" rev-parse --short HEAD)" = b677c39 ] || { echo "SphericalRemesh pin mismatch" >&2; exit 1; }
[ -f "$ORA/_dl/CLI11.hpp" ] || { echo "run build_spherical_remesh.sh first (CLI11 cache)" >&2; exit 1; }
CXX=${CXX:-/usr/bin/clang++}
mkdir -p "$OUT/bin"
{
  echo "# $($CXX --version | head -1)"
  set -x
  $CXX -std=c++11 -O2 -DNDEBUG -ffp-contract=off -w -include cstddef -DSREMESH_VERSION=\"v1.2.1\" \
    -I"$SR/src" -I"$SR/wrapper" -I"$ORA/_dl" -I"$ML/include" -I"$ML/include/Util" \
    "$SR/src/SphericalRemeshing.cpp" "$SR/wrapper/main.cpp" \
    "$ML/src/Mesh.cpp" "$ML/src/MeshIO.cpp" "$ML/src/MNIObjIO.cpp" "$ML/src/VtkIO.cpp" \
    "$ML/src/Util/AABB_Sphere.cpp" "$ML/src/Util/AABB.cpp" "$ML/src/Util/Geom.cpp" \
    "$ML/src/Util/SphericalHarmonics.cpp" "$ML/src/Util/SurfaceUtil.cpp" \
    -o "$OUT/bin/SphericalRemesh"
} > "$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
echo "OK: $OUT/bin/SphericalRemesh"
