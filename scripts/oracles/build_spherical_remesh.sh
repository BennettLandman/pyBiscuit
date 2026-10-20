#!/bin/bash
# Build upstream SphericalRemesh CLI with plain clang++ (mirrors its CMakeLists.txt).
# Requires oracles/meshlib (run build_meshlib.sh first) and CLI11 v1.9.1 single header
# (CMake file(DOWNLOAD)s it; we cache it at oracles/_dl/CLI11.hpp, sha256 recorded below).
# Output: $ORACLES/spherical-remesh/{bin/SphericalRemesh,build.log}
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${BISCUIT_ROOT:-$HOME/dev/biscuit}"
ORA="${ORACLES:-$ROOT/oracles}"
SRC="$ROOT/reference/repos/SphericalRemesh"
OUT="$ORA/spherical-remesh"
PIN=b677c39
CLI11_URL=https://github.com/CLIUtils/CLI11/releases/download/v1.9.1/CLI11.hpp
got=$(git -C "$SRC" rev-parse --short HEAD)
[ "$got" = "$PIN" ] || { echo "SphericalRemesh commit mismatch: $got != $PIN" >&2; exit 1; }
[ -f "$ORA/meshlib/lib/libMesh.a" ] || { echo "build MeshLib first" >&2; exit 1; }
mkdir -p "$ORA/_dl"
[ -f "$ORA/_dl/CLI11.hpp" ] || curl -sSL -m 120 -o "$ORA/_dl/CLI11.hpp" "$CLI11_URL"
CXX=${CXX:-/usr/bin/clang++}
CXXFLAGS=${CXXFLAGS:--std=c++11 -O2 -DNDEBUG}
VER=1.2.1   # SREMESH_VERSION_{MAJOR,MINOR,PATCH} in CMakeLists.txt
rm -rf "$OUT"; mkdir -p "$OUT/bin" "$OUT/src"
cp -R "$SRC/src" "$SRC/wrapper" "$OUT/"
shopt -s nullglob
for p in "$HERE"/patches/spherical*.patch; do echo "applying $p"; (cd "$OUT" && patch -p1 < "$p"); done
M="$ORA/meshlib/include"
{
  echo "# $($CXX --version | head -1)"; echo "# commit $got; CLI11 sha256 $(shasum -a 256 "$ORA/_dl/CLI11.hpp" | cut -d' ' -f1)"
  set -x
  $CXX $CXXFLAGS -DSREMESH_VERSION=\"v$VER\" -I"$OUT/src" -I"$OUT/wrapper" -I"$ORA/_dl" -I"$M" -I"$M/Util" \
    "$OUT/src/SphericalRemeshing.cpp" "$OUT/wrapper/main.cpp" "$ORA/meshlib/lib/libMesh.a" -o "$OUT/bin/SphericalRemesh"
} > "$OUT/build.log" 2>&1
grep -q xcrun_db "$OUT/build.log" && true
[ -x "$OUT/bin/SphericalRemesh" ] || { echo "build failed, see $OUT/build.log" >&2; exit 1; }
echo "OK: $OUT/bin/SphericalRemesh"
