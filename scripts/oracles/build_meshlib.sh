#!/bin/bash
# Build upstream MeshLib (static libMesh.a) with plain clang++ (no cmake).
# Mirrors reference/repos/MeshLib/CMakeLists.txt: C++11, all SRC/UTIL/GEOD sources.
# Output: $ORACLES/meshlib/{lib/libMesh.a,include/,build.log,READY}
# Pinned commit: dcef807 (checked, not checked out).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${BISCUIT_ROOT:-$HOME/dev/biscuit}"
SRC="$ROOT/reference/repos/MeshLib"
OUT="${ORACLES:-$ROOT/oracles}/meshlib"
PIN=dcef807
got=$(git -C "$SRC" rev-parse --short HEAD)
[ "$got" = "$PIN" ] || { echo "MeshLib commit mismatch: $got != $PIN" >&2; exit 1; }
CXX=${CXX:-/usr/bin/clang++}
CXXFLAGS=${CXXFLAGS:--std=c++11 -O2 -DNDEBUG -ffp-contract=off}
rm -rf "$OUT"; mkdir -p "$OUT/lib" "$OUT/obj"
# include/ is a copy so patches never touch reference/. Patches: patches/meshlib*.patch (build-only).
cp -R "$SRC/include" "$OUT/include"; cp -R "$SRC/src" "$OUT/src"
shopt -s nullglob
for p in "$HERE"/patches/meshlib*.patch; do
  echo "applying $p"; (cd "$OUT" && patch -p1 < "$p")
done
INC="-I$OUT/include -I$OUT/include/Geodesic -I$OUT/include/Geodesic/gw -I$OUT/include/Geodesic/gw/gw_core -I$OUT/include/Geodesic/gw/gw_geodesic -I$OUT/include/Geodesic/gw/gw_maths -I$OUT/include/Util"
{
  echo "# $($CXX --version | head -1)"; echo "# commit $got"; echo "# flags: $CXXFLAGS $INC"
  cd "$OUT"
  # exactly the sources of CMakeLists.txt (SRC_FILES, SRC_UTIL_FILES, SRC_GEOD_FILES)
  grep -o 'src/[A-Za-z0-9_/]*\.cpp' "$SRC/CMakeLists.txt" > obj/sources.txt
  i=0
  while read -r f; do
    o="obj/$(echo "${f%.cpp}" | tr / _).o"
    $CXX $CXXFLAGS $INC -c "$f" -o "$o" 2>"$o.err" &
    i=$((i+1)); if [ $((i%8)) -eq 0 ]; then wait; fi
  done < obj/sources.txt
  wait
  cat obj/*.err
  if cat obj/*.err | grep ' error' | grep -qv xcrun_db; then echo 'COMPILE ERRORS' ; fi
  ar rcs lib/libMesh.a obj/*.o
  ranlib lib/libMesh.a
} > "$OUT/build.log" 2>&1
! grep -q 'COMPILE ERRORS' "$OUT/build.log" || { echo "compile errors, see $OUT/build.log" >&2; exit 1; }
[ -s "$OUT/lib/libMesh.a" ] || { echo "build failed, see $OUT/build.log" >&2; exit 1; }
echo "libMesh.a built from MeshLib $got; include=$OUT/include (+Util, Geodesic/...)" > "$OUT/READY"
echo "OK: $OUT/lib/libMesh.a"
