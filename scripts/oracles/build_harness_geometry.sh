#!/bin/bash
# Build the B2.4a oracle driver against the O1a MeshLib: $ORACLES/meshlib/bin/harness_geometry.
# Needs $ORACLES/meshlib/READY (run build_meshlib.sh first). Default flags (arm64 clang fuses
# multiply-adds); tests/data/golden/meshlib/geometry_* use -ffp-contract=off, see make_golden_geometry.py.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${BISCUIT_ROOT:-$HOME/dev/biscuit}"
OUT="${ORACLES:-$ROOT/oracles}/meshlib"
[ -f "$OUT/READY" ] || { echo "run build_meshlib.sh first" >&2; exit 1; }
CXX=${CXX:-/usr/bin/clang++}
mkdir -p "$OUT/bin"
$CXX -std=c++11 -O2 -DNDEBUG -I"$OUT/include" -I"$OUT/include/Util" "$HERE/harness_geometry.cpp" \
  "$OUT/lib/libMesh.a" -o "$OUT/bin/harness_geometry"
echo "$OUT/bin/harness_geometry"
